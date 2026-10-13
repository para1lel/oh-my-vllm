"""A cancelled oneoff capture must stop handlers before freezing raw evidence."""

import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts.acceptance_proxy import recording_proxy


@pytest.mark.parametrize("headers_sent", [False, True])
def test_proxy_close_joins_owned_header_and_body_readers(tmp_path, headers_sent):
    reached, release = threading.Event(), threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if headers_sent:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": waiting\n\n")
                self.wfile.flush()
            reached.set()
            release.wait(5)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.daemon_threads = True
    server_thread = threading.Thread(target=upstream.serve_forever)
    server_thread.start()
    client_errors = []

    def request(url):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url + "/chat/completions", data=b"{}"),
                timeout=5,
            ) as response:
                response.read()
        except (OSError, urllib.error.URLError) as error:
            client_errors.append(str(error))

    client = None
    try:
        with recording_proxy(
            f"http://127.0.0.1:{upstream.server_port}/v1",
            raw_directory=tmp_path / "wire",
        ) as (url, records):
            client = threading.Thread(target=request, args=(url,))
            client.start()
            assert reached.wait(3)
        client.join(3)
        assert not client.is_alive()
        assert len(records) == 1
        record = records[0]
        assert record["response_sha256"]
        assert (tmp_path / "wire" / record["response_file"]).is_file()
        # Returning from the context joins the handler that owns these fields.
        assert "response_ids" in record
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        server_thread.join(3)
        if client:
            client.join(3)

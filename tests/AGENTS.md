# Service tests

Use the Python standard library test runner and the Rust service executable.
Start the Python worker through the service process.

Use a different IPC path, port, and process group for each test.
Stop the test processes after each test. Remove the temporary files.

Do tests of the outputs at the process boundary.
Use the same request seed when you compare random outputs.

Do not use placeholder execution time as model inference time.
Before you add a test, tell the user why it is necessary.

Do not save test reports or result records.

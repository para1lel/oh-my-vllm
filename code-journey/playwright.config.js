import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: process.env.JOURNEY_URL || "http://127.0.0.1:18084",
    browserName: "chromium", viewport: { width: 1536, height: 1024 }, colorScheme: "light" },
});

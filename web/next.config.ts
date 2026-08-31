import type { NextConfig } from "next";

const API = process.env.KINDRED_API_URL ?? "http://127.0.0.1:8317";

const nextConfig: NextConfig = {
  output: "standalone",
  // Next 16 blocks /_next/* from 127.0.0.1 unless listed; the preview
  // browser uses that host, so without this the app SSRs then never hydrates.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;

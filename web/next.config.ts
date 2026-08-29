import type { NextConfig } from "next";

const API = process.env.KINDRED_API_URL ?? "http://127.0.0.1:8317";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;

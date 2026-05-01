/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow other devices on the same Wi-Fi to load the dev server
  // without the cross-origin warning. Next 15 will hard-require this.
  // Add new IPs/hostnames here if your DHCP lease changes.
  allowedDevOrigins: [
    "192.168.1.41",
    "localhost",
  ],
};

export default nextConfig;

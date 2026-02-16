# RAG-Bench Frontend Dockerfile
# Multi-stage build for minimal image size

# Stage 1: Build
FROM node:20-alpine as builder

WORKDIR /build

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Serve
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy built application from builder
COPY --from=builder /build/dist /usr/share/nginx/html

# Create a non-root user for nginx
RUN addgroup -g 101 -S nginx 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

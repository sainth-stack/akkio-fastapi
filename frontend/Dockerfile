FROM node:14

WORKDIR /app

COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Increase Node.js memory limit
ENV NODE_OPTIONS="--max-old-space-size=4096"

# Build the application
RUN npm run build

# Expose the app's port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]

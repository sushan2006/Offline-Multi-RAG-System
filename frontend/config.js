// 🌐 RAG Assistant Configuration
// This file is used to store global settings for the frontend.

const API_CONFIG = {
    // When running locally, this is http://127.0.0.1:8000
    // When deploying via Tunnel, our startup script will automatically update this.
    BASE_URL: "http://127.0.0.1:8000"
};

// Export for use in HTML scripts
window.API_CONFIG = API_CONFIG;

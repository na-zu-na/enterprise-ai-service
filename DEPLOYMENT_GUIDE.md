# Public Course Deployment Guide

This repository has two public course-demo entry points.

1. **GitHub Pages** serves `docs/index.html`. It is a static, always-available fallback that demonstrates knowledge-base scoping, grounded citations, cancellation, retry, and human approval.
2. **Render** runs `demo_backend/app.py`. It exposes the same course-demo flow via a live FastAPI API. It deliberately does not require or expose the private Spring service, production database, Calendar MCP connection, or model credentials.

## Publish GitHub Pages

1. Push this branch to GitHub.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the branch that contains these files (normally `main`) and the `/docs` folder, then save.
5. GitHub shows the public URL, normally `https://na-zu-na.github.io/enterprise-ai-service/`.

## Deploy the live backend on Render

1. Sign in to Render and select **New → Blueprint**.
2. Connect this GitHub repository and select `render.yaml`.
3. Create the `pe6203-enterprise-assistant-demo` service on the Free plan.
4. Wait for deployment to complete, then open `/health` on the generated `https://...onrender.com` URL to confirm the service is available.
5. The GitHub Pages interface is configured to use the deployed Render URL by default. Users can still enter a replacement URL or select **Use built-in demo** when needed.

The Render free service can sleep after inactivity. The static GitHub Pages demo remains available if the live API needs time to start or is unavailable. Do not add secrets to this demo service.

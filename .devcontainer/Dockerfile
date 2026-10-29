FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Fixed GID across host, devcontainer, and job containers so bind-mounted data stays group-writable.
ARG SHARED_DATA_GID=1500
RUN groupadd -g "${SHARED_DATA_GID}" shareddata \
    && usermod -aG shareddata vscode

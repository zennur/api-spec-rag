# API Marketplace

A powerful RAG (Retrieval-Augmented Generation) application for discovering and searching OpenAPI specifications across multiple sources.

## Overview

API Marketplace is an intelligent search platform that aggregates OpenAPI specifications from various sources including APIGuru and GitHub, making it easy for developers to discover and explore APIs that match their needs.

## Features

### 🔍 Comprehensive API Discovery
- **Multi-Source Crawling**: Automatically crawls OpenAPI specs from:
  - APIGuru directory
  - GitHub repositories
  - Custom API documentation sources
- **Swagger Documentation**: Extracts and indexes complete Swagger/OpenAPI documentation

### 🎯 Intelligent Search
- **RAG-Powered Search**: Leverages Retrieval-Augmented Generation for semantic API discovery
- **Natural Language Queries**: Find APIs using plain English descriptions of what you need
- **Precise Results**: Get relevant API endpoints and specifications based on your requirements

### 🏢 Enterprise-Ready
- **Private RAG Instances**: Build smaller-scale RAG systems with your own curated API documentation
- **Enhanced Developer Experience**: Provide your team with instant access to internal and external APIs
- **Custom Collections**: Organize APIs by domain, team, or project

## Use Cases

### For Developers
- Quickly find APIs that match specific functionality requirements
- Explore API capabilities without manually browsing documentation
- Discover alternative APIs for your use case

### For Organizations
- Create internal API marketplaces for better API governance
- Improve developer onboarding with searchable API documentation
- Reduce time spent searching for the right API

### For API Providers
- Increase API discoverability
- Improve developer adoption
- Provide better documentation experience

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/api-spec-rag.git
cd api-spec-rag

# Install dependencies
npm install

# Configure your API sources
cp .env.example .env

# Start crawling and indexing
npm run crawl

# Launch the search interface
npm start
```

## Architecture
The application consists of three main components:

1. Crawler: Discovers and downloads OpenAPI specifications from configured sources
2. Indexer: Processes and indexes API documentation for efficient retrieval
3. Search Engine: RAG-powered search interface for querying the indexed APIs

## Configuration
Create a .env file with your configuration:

```bash
# Data sources
APIGURU_ENABLED=true
GITHUB_ENABLED=true
GITHUB_TOKEN=your_github_token

# RAG Configuration
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-4

# Search Settings
MAX_RESULTS=10
SIMILARITY_THRESHOLD=0.7
```

## Custom RAG Instances
Build your own private API marketplace:

```bash
# Add your OpenAPI specs to the custom directory
mkdir -p data/custom-apis
cp your-api-spec.yaml data/custom-apis/

# Index only custom APIs
npm run index -- --source custom

# Deploy your private instance
npm run deploy
```

## Contributing
Contributions are welcome! Please read our Contributing Guide for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
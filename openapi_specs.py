import requests
import os
import yaml

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # set this in your env for higher rate limits
SESSION = requests.Session()
if GITHUB_TOKEN:
    SESSION.headers.update({"Authorization": f"Bearer {GITHUB_TOKEN}"})
SESSION.headers.update({"Accept": "application/vnd.github.v3+json"})

def get_openapi_specs_apiguru():

    # Fetch the APIs.guru list
    response = requests.get('https://api.apis.guru/v2/list.json')
    apis = response.json()

    # Extract all OpenAPI spec URLs
    spec_urls = []
    for provider, data in apis.items():
        for version, details in data.get('versions', {}).items():
            # Get both JSON and YAML URLs
            swagger_url = details.get('swaggerUrl')
            swagger_yaml_url = details.get('swaggerYamlUrl')
            title = details.get('info', {}).get('title', 'N/A')
            
            spec_urls.append({
                'provider': provider,
                'version': version,
                'title': details.get('info', {}).get('title', 'N/A'),
                'json_url': swagger_url,
                'yaml_url': swagger_yaml_url,
                'openapi_version': details.get('openapiVer', 'N/A')
            })
            extract_specs(swagger_yaml_url,title.replace("/","_"))

    print(f"Found {len(spec_urls)} OpenAPI specifications")
    
    text_file = open(f"apiguruopendocs.md", "w")
    text_file.write(str(spec_urls))
    
    return spec_urls


def get_openapi_specs_github(query, max_results=200):
    """
    Search GitHub code for OpenAPI specs and return a list of dicts with useful URLs.
    query: GitHub code search query string (see examples below)
    max_results: cap on number of results to fetch
    """
    results = []
    per_page = 100
    page = 1
    while len(results) < max_results:
        params = {"q": query, "per_page": per_page, "page": page}
        r = SESSION.get("https://api.github.com/search/code", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            break

        for it in items:
            repo = it["repository"]
            owner_repo = repo["full_name"]  # e.g., "org/repo"
            path = it["path"]               # file path in repo
            sha = it["sha"]                 # blob SHA
            # raw URL via blob SHA (avoids needing default branch name)
            raw_url = f"raw.githubusercontent.com/{owner_repo}/{sha}/{path}"
            results.append({
                "name": it.get("name"),
                "path": path,
                "repo": owner_repo,
                "html_url": it.get("html_url"),
                "raw_url": raw_url,
            })
            if len(results) >= max_results:
                break
            filename = owner_repo.split('/')[1]
            crawl_url = it.get("html_url").replace("github", "raw.githubusercontent").replace("/blob", "")
            print(it.get("html_url"))
            print(crawl_url)
            extract_specs(crawl_url, filename)
        page += 1
        
    return results

def extract_specs(crawl_url, filename):
    with open(f"api_docs/{filename}.yaml", "wb") as f:
        f.write(requests.get(crawl_url).content)


if __name__ == "__main__":
    # apis = get_openapi_specs_list()
    # print(apis)

    # Example queries (pick one)
    queries = [
        'filename:openapi.yaml',
        'filename:openapi.yml',
        'filename:openapi.json',
        '"openapi: 3" extension:yml',
        '"openapi: 3" extension:yaml',
        '"openapi": "3" extension:json',
        'filename:swagger.json',
        '"swagger: \\"2.0\\"" extension:yaml',
    ]

    # Run one query
    # specs = get_openapi_specs_github(queries[0], max_results=300)
    # print(f"Found {len(specs)} files")
    # for s in specs[:10]:
    #     print(f"- {s['repo']}/{s['path']}\n  raw: {s['raw_url']}")

    # text_file = open(f"githubopendocs.md", "w")
    # text_file.write(str(specs))

    get_openapi_specs_apiguru()

    
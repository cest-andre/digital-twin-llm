import json
import requests


syndeia_prompt_template = "You are a friendly virtual assistant who will ask questions regarding artifact relations in a Syndeia server.  You will be provided context of relations that may be relevant to answer the question, although this is not guaranteed.  Ensure your answer reflects what is provided in the context.  IMPORTANT:  only address matters related to Syndeia.  Politely decline to answer about unrelated topics such as coding, current events, pop culture, history, science, etc.  \
The context will contain a list of Syndeia artifact relations.  The format is the key, name, and relations count of two artifacts with '<-->' between them which symbolizes a relation they share.  The relations count is how many relations the artifact has across the entire database.  \
Here is an example of a relation between Artifact NCAM-xx and NCAM-yy:  Relation: (key: NCAM-xx, name: Some Name, relations_count: some integer) <--> (key: NCAM-yy, name: Some Different Name, relations_count: another integer) \
RULES:  1)  Do not respond by copying the context.  Reform relevant context and answer using only natural language.  2)  Keep your answers brief; simply answer the question without providing additional commentary.  3)  If relevant relations are not provided in the context, then report that in your response and end there.  4)  If relation counts are asked, use the relations_count value found where that artifact appears in the context.  \
Begin!\n\nPrevious conversation history: {chat_history}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer: "


def get_syndeia_relations(write_path=None):
    signIn = ''

    username = ''
    pw = ''

    creds = {'username': username, 'password': pw, 'rememberMe': True}
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    response = requests.post(signIn, json=creds, headers=headers)

    assert response.status_code == 200

    token = response.json()
    x_auth_token = token['resources']['token']

    get_url = 'http://thread.ncam-dt.com/repositories'
    headers = {
        'Accept': 'application/json',
        'X-Auth-Token': x_auth_token
    }

    get_response = requests.get(get_url, headers=headers)
    assert response.status_code == 200

    # Extract the resources list from the response JSON
    repos = get_response.json()["resources"]

    syndeia_data = []
    art_keys = {}

    for repo in repos:
        if "internal" in repo and "key" in repo["internal"]:
            repo_key = repo["internal"]["key"]

            get_containers = f'http://thread.ncam-dt.com/containers?repository.key={repo_key}'
            response = requests.get(get_containers, headers=headers)
            assert response.status_code == 200

            containers = response.json()["resources"]
            for container in containers:
                if "internal" in container and "key" in container["internal"]:
                    container_key = container["internal"]["key"]
                    get_relations = f'http://thread.ncam-dt.com/relations?container.key={container_key}&perPage=1000000'
                    response = requests.get(get_relations, headers=headers)
                    assert response.status_code == 200

                    relations = response.json()["resources"]

                    for relation in relations:
                        relation_structure = {
                            "relationContainer": relation["container"]
                        }
                        art_key = relation['sourceArtifact']['internal']['key']
                        get_artifact = f"http://thread.ncam-dt.com/artifacts/{art_key}"
                        response = requests.get(get_artifact, headers=headers)
                        assert response.status_code == 200

                        artifact = response.json()["resources"]
                        artifact = {
                            "internalKey": artifact["internal"]["key"],
                            "externalKey": artifact["external"]["key"],
                            "version": artifact["version"],
                            "name": artifact["name"],
                            "createdBy": artifact["createdBy"]["internal"]["key"],
                            "createdDate": artifact["createdDate"],
                            "otherInfo": artifact["otherInfo"]
                        }

                        relation_structure["sourceArtifact"] = artifact

                        if art_key not in art_keys.keys():
                            art_keys[art_key] = get_relation_count(headers, art_key)

                        art_key = relation['targetArtifact']['internal']['key']
                        get_artifact = f"http://thread.ncam-dt.com/artifacts/{relation['targetArtifact']['internal']['key']}"
                        response = requests.get(get_artifact, headers=headers)
                        assert response.status_code == 200

                        artifact = response.json()["resources"]
                        artifact = {
                            "internalKey": artifact["internal"]["key"],
                            "externalKey": artifact["external"]["key"],
                            "version": artifact["version"],
                            "name": artifact["name"],
                            "createdBy": artifact["createdBy"]["internal"]["key"],
                            "createdDate": artifact["createdDate"],
                            "otherInfo": artifact["otherInfo"]
                        }

                        relation_structure["targetArtifact"] = artifact

                        if art_key not in art_keys.keys():
                            art_keys[art_key] = get_relation_count(headers, art_key)

                        syndeia_data.append(f'Relation: (key: {relation_structure["sourceArtifact"]["externalKey"]}, name: {relation_structure["sourceArtifact"]["name"]}, relations_count: {art_keys[relation_structure["sourceArtifact"]["internalKey"]]}) <--> (key: {relation_structure["targetArtifact"]["externalKey"]}, name: {relation_structure["targetArtifact"]["name"]}, relations_count: {art_keys[relation_structure["targetArtifact"]["internalKey"]]})')          
                
    sydeia_json = json.dumps(syndeia_data, indent=None)

    if write_path is not None:
        with open(write_path, "w") as outfile:
            outfile.write(sydeia_json)

    return syndeia_data


def get_relation_count(headers, key):
    #   Get relation count for artifact.
    get_relation_count = f"http://thread.ncam-dt.com/relations?source.key={key}&perPage=1000000"
    response = requests.get(get_relation_count, headers=headers)
    assert response.status_code == 200

    relation_list = response.json()["resources"]
    count = len(relation_list)

    get_relation_count = f"http://thread.ncam-dt.com/relations?target.key={key}&perPage=1000000"
    response = requests.get(get_relation_count, headers=headers)
    assert response.status_code == 200

    relation_list = response.json()["resources"]
    count += len(relation_list)
    
    return count
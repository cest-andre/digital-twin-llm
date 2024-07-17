import boto3
import json
import os
import io
from pypdf import PdfReader


datacat_prompt_template = "You are a friendly virtual assistant who will ask questions regarding assets in our data catalog.  Based on the user's question, a context will be provided that may contain the answer.  Ensure your answer reflects what is provided in the context.  IMPORTANT:  only address matters related to the catalog.  Politely decline to answer about unrelated topics such as coding, current events, pop culture, history, science, etc.  \
The context will contain a list of asset content chunks that hopefully obtains what the user requests.  Content for a particular asset may consistent of multiple chunks, which all share the same asset_id.  \
RULES:  1)  Do not respond by copying the context.  Reform relevant context and answer using only natural language.  2)  Keep your answers brief; simply answer the question without providing additional commentary.  3)  If relevant information are not provided in the context, then report that in your response and end there.  \
Begin!\n\nPrevious conversation history: {chat_history}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer: "


def get_datacat_assets(write_path=None, chunk_interval=1000):
    sts = boto3.client('sts', region_name='us-gov-west-1', aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    token_mfa_code = input("Enter MFA code:  ")

    sess_tok = sts.get_session_token(
        DurationSeconds=1000,
        SerialNumber='',
        TokenCode=token_mfa_code
    )

    bucket_name = "ncam-shared-3d-models-bucket"

    client = boto3.client(
        's3',
        region_name='us-gov-west-1',
        aws_access_key_id=sess_tok['Credentials']['AccessKeyId'],
        aws_secret_access_key=sess_tok['Credentials']['SecretAccessKey'],
        aws_session_token=sess_tok['Credentials']['SessionToken']
    )

    s3 = boto3.resource(
        's3',
        region_name='us-gov-west-1',
        aws_access_key_id=sess_tok['Credentials']['AccessKeyId'],
        aws_secret_access_key=sess_tok['Credentials']['SecretAccessKey'],
        aws_session_token=sess_tok['Credentials']['SessionToken']
    )

    manifest_obj = s3.Bucket(bucket_name).Object("manifest/manifest.json")
    manifest_obj = manifest_obj.get()['Body'].read().decode('utf-8')
    manifest_obj = json.loads(manifest_obj)

    asset_data = []
    for asset_id in manifest_obj.keys():
        metadata_obj = s3.Bucket(bucket_name).Object(f"BIM/Assets/{asset_id}/meta.json").get()

        metadata_body = metadata_obj['Body'].read().decode('utf8')
        metadata_body = json.loads(metadata_body)
        metadata_body.pop("id")

        asset_metadata = f'asset_id: {asset_id}, body: {json.dumps(metadata_body)}'
        asset_data.append(asset_metadata)

        contents = client.list_objects(Bucket=bucket_name, Prefix=f"BIM/Assets/{asset_id}/", Delimiter='/')

        if 'CommonPrefixes' not in contents.keys():
            continue

        #   contents['CommonPrefixes'] lists subdirs in directory.
        for prefix in contents['CommonPrefixes']:
            asset_subdir = prefix['Prefix']

            #   TODO:  Which subdirs to check?
            if 'documentation' in asset_subdir:
                doc_files = client.list_objects(Bucket=bucket_name, Prefix=asset_subdir, Delimiter='/')['Contents']

                for doc in doc_files:
                    doc_obj = s3.Bucket(bucket_name).Object(doc['Key']).get()

                    # print(doc_obj['ContentLength'])

                    doc_text = None
                    if doc_obj['ContentType'] == 'application/pdf':
                        pdf_bytes = doc_obj['Body'].read()
                        byte_reader = io.BytesIO(pdf_bytes)

                        body_pdf = PdfReader(byte_reader)
                        
                        #   is text splitting needed if parsing by pdf page?
                        #   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        #   texts = text_splitter.split_documents(docs)
                        all_text = ""
                        for page in body_pdf.pages:
                            body_page = page.extract_text()

                            # does stripping reduce memory required per chunk?
                            body_page = body_page.replace('\n', ' ')

                            all_text += body_page

                        doc_text = all_text

                    elif doc_obj['ContentType'] == 'text/csv':
                        csv_body = doc_obj['Body'].read().decode('utf8')
                        doc_text = csv_body

                    else:
                        print(f"CONTENT TYPE NOT SUPPORTED:  {doc_obj['ContentType']}")
                        continue

                    for i in range(0, len(doc_text), chunk_interval):
                        chunk = doc_text[i:i+chunk_interval]

                        if len(chunk) > 0:
                            asset_entry = f'asset_id: {asset_id}, body: {chunk}'
                            asset_data.append(asset_entry)

            if 'Revit' in asset_subdir:
                doc_obj = s3.Bucket(bucket_name).Object(asset_subdir + 'DatasmithSev1Office_Metadata.json').get()['Body'].read().decode('utf-8')
                doc_obj = json.loads(doc_obj)

                for elm in doc_obj['Elements']:
                    revit_metadata = {"ElementId": elm['ElementId'], "Category": elm['Category'], "Properties": {}}
                    for k in elm['Properties'].keys():
                        if elm['Properties'][k] is not None and 'None' not in elm['Properties'][k]:
                            revit_metadata['Properties'][k] = elm['Properties'][k]

                    revit_metadata = json.dumps(revit_metadata).replace('"', '').replace('\\', '')
                    revit_metadata = f'asset_id: {asset_id}, body: {revit_metadata}'
                    asset_data.append(revit_metadata)

    asset_json = json.dumps(asset_data, indent=None)

    if write_path is not None:
        with open(write_path, "w") as outfile:
            outfile.write(asset_json)

    return asset_data
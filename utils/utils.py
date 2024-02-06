from langchain_community.document_loaders import JSONLoader
import json


def save_to_json(transcript,filename):
    with open(filename, 'w') as f:
        json.dump(transcript, f, indent=4)

def load_documents(filepath):
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["start"] = record.get("start")
        if "end" in record.keys():
            metadata["end"] = record.get("end")
        return metadata

    loader = JSONLoader(
        file_path=filepath,
        jq_schema='.[]',
        text_content=True,
        content_key="text",
        metadata_func = metadata_func
    )

    documents = loader.load()

    return documents
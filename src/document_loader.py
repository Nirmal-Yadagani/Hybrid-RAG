from pathlib import Path

import yaml
from langchain_core.documents import Document


def load_docs():

    with open('config.yaml', 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    folder_path = Path(params['raw_data_folderpath'])

    documents = []

    for file_path in folder_path.rglob("*.txt"): # type: ignore

        try:
            with open(file_path, 'r') as f:
                text_content = f.read()

                document = Document(page_content=text_content,
                                    metadata={
                                        'source':file_path.name,
                                        'category':file_path.name
                                    })

                documents.append(document)

        except Exception as e:  # noqa: BLE001
            print(f'Following error occured: {e}')

    print(f"Loaded {len(documents)} text files.")
    return documents


def main():
    load_docs()


if __name__=='__main__':
    main()
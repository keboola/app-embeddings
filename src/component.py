import csv
import logging
import hashlib

import pyarrow as pa
import pandas as pd

from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException
from configuration import Configuration

from openai import OpenAI

class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None
        self.client = None

    def run(self):
        self.init_configuration()
        self.init_openai_client()

        try:
            input_table = self._get_input_table()
            with open(input_table.full_path, 'r', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)
                self._process_rows_csv(reader)
        except Exception as e:
            raise UserException(f"Error occurred during embedding process: {str(e)}")


    def _get_linking_table(self):
        destination_config = self.configuration.parameters['destination']
        base_output_name = destination_config.get("output_table_name", "openAI-embedding")
        linking_table_name = f"{base_output_name}-linking.csv"
        return self.create_out_table_definition(linking_table_name)

    def _process_rows_csv(self, reader):
        output_table = self._get_output_table()
        linking_table = self._get_linking_table()

        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file, \
            open(linking_table.full_path, 'w', encoding='utf-8', newline='') as linking_file:
            
            fieldnames = reader.fieldnames + ['embedding', 'parent_id']
            linking_fieldnames = ['parent_id', self._configuration.embed_column]

            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            linking_writer = csv.DictWriter(linking_file, fieldnames=linking_fieldnames)

            writer.writeheader()
            linking_writer.writeheader()

            self.row_count = 0
            if self._configuration.chunking.is_enabled:
                self.chunk_process_rows_csv(reader, linking_writer)
            else:
                for row in reader:
                    self.row_count += 1
                    text = row[self._configuration.embed_column]
                    parent_id = self.generate_hash(text)
                    embedding = self.get_embedding(text, model=self._configuration.model)

                    row['embedding'] = embedding if embedding else "[]"
                    row['parent_id'] = parent_id
                    writer.writerow(row)

                    # Write to linking table
                    linking_writer.writerow({'parent_id': parent_id, self._configuration.embed_column: text})


    def chunk_process_rows_csv(self, reader, linking_writer):
        chunk_size = self._configuration.chunking.size
        chunk_method = self._configuration.chunking.method

        output_table = self._get_output_table()

        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
            fieldnames = reader.fieldnames + ['embedding', 'parent_id']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            self.row_count = 0
            for row in reader:
                text = row[self._configuration.embed_column]
                parent_id = self.generate_hash(text)
                linking_writer.writerow({'parent_id': parent_id, self._configuration.embed_column: text})  # Linking entry

                chunks = []
                if chunk_method == "words":
                    words = text.split()
                    for i in range(0, len(words), chunk_size):
                        chunks.append(" ".join(words[i:i + chunk_size]))
                elif chunk_method == "characters":
                    for i in range(0, len(text), chunk_size):
                        chunks.append(text[i:i + chunk_size])

                for chunk in chunks:
                    embedding = self.get_embedding(chunk, model=self._configuration.model)
                    row_copy = row.copy()
                    row_copy['embedding'] = embedding if embedding else "[]"
                    row_copy[self._configuration.embed_column] = chunk
                    row_copy['parent_id'] = parent_id
                    writer.writerow(row_copy)
                    self.row_count += 1

    def generate_hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

    def init_openai_client(self):
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)

    def get_embedding(self, text, model):
        if not text or not isinstance(text, str) or text.strip() == "":
                return []
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding

    def _get_input_table(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")
        return self.get_input_tables_definitions()[0]

    def _get_output_table(self):
        destination_config = self.configuration.parameters['destination']
        if not (out_table_name := destination_config.get("output_table_name")):
            out_table_name = f"openAI-embedding.csv"
        else:
            out_table_name = f"{out_table_name}.csv"

        return self.create_out_table_definition(out_table_name)
        

if __name__ == "__main__":
    try:
        comp = Component()
        comp.execute_action()
        logging.getLogger().setLevel(logging.WARNING)
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
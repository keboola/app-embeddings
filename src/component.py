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

        batch_size = 10
        batch_data = []
        linking_data = []

        fieldnames = reader.fieldnames + ['embedding', 'parent_id']
        linking_fieldnames = ['parent_id', self._configuration.embed_column]

        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file, \
             open(linking_table.full_path, 'w', encoding='utf-8', newline='') as linking_file:
            
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            linking_writer = csv.DictWriter(linking_file, fieldnames=linking_fieldnames)

            writer.writeheader()
            linking_writer.writeheader()

            for row in reader:
                text = row[self._configuration.embed_column]
                parent_id = self.generate_hash(text)

                if self._configuration.chunking.is_enabled:
                    chunked_texts = self.chunk_text(text)
                    for chunk in chunked_texts:
                        batch_data.append((row.copy(), chunk, parent_id))

                else:
                    batch_data.append((row, text, parent_id))

                linking_data.append({'parent_id': parent_id, self._configuration.embed_column: text})

                # Process batch when it reaches batch size
                if len(batch_data) >= batch_size:
                    self._process_batch(batch_data, writer)
                    batch_data.clear()

            # Process remaining batch if any
            if batch_data:
                self._process_batch(batch_data, writer)

            # Write linking table data in batch
            linking_writer.writerows(linking_data)

    def _process_batch(self, batch_data, writer):
        texts = [text for _, text, _ in batch_data]
        embeddings = self.get_batch_embeddings(texts)

        for i, (row, text, parent_id) in enumerate(batch_data):
            row['embedding'] = embeddings[i] if embeddings[i] else "[]"
            row['parent_id'] = parent_id
            row[self._configuration.embed_column] = text
            writer.writerow(row)

    def chunk_text(self, text):
        """
        Splits text into smaller chunks based on the configured method.
        """
        chunk_size = self._configuration.chunking.size
        chunk_method = self._configuration.chunking.method

        chunks = []
        if chunk_method == "words":
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))
        elif chunk_method == "characters":
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])
        else:
            chunks.append(text)  # Default: No chunking

        return chunks

    def get_batch_embeddings(self, texts):
        """
        Calls OpenAI embedding API in batch mode.
        """
        try:
            response = self.client.embeddings.create(input=texts, model=self._configuration.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logging.error(f"Failed batch embedding: {e}")
            return [[] for _ in texts]  # Return empty embeddings in case of failure

    def generate_hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

    def init_openai_client(self):
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)

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

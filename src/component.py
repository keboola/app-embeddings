import csv
import logging

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

    def _process_rows_csv(self, reader):
        output_table = self._get_output_table()
        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
            fieldnames = reader.fieldnames + ['embedding']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            self.row_count = 0
            if self._configuration.chunkingEnabled:
                self.chunk_process_rows_csv(reader)
            else:
                for row in reader:
                    self.row_count += 1
                    text = row[self._configuration.embedColumn]
                    embedding = self.get_embedding(text)
                    row['embedding'] = embedding if embedding else "[]" # handles empty embeddings
                    writer.writerow(row)

    def chunk_process_rows_csv(self, reader):
        chunk_size = self._configuration.chunkSize
        chunk_method = self._configuration.chunkMethod
        output_table = self._get_output_table()

        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
            fieldnames = reader.fieldnames + ['embedding']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            self.row_count = 0
            for row in reader:
                text = row[self._configuration.embedColumn]
                chunks = []
                if chunk_method == "words":
                    words = text.split()
                    for i in range(0, len(words), chunk_size):
                        chunks.append(" ".join(words[i:i + chunk_size]))
                elif chunk_method == "characters":
                    for i in range(0, len(text), chunk_size):
                        chunks.append(text[i:i + chunk_size])
                        
                for chunk in chunks:
                    embedding = self.get_embedding(chunk)
                    row_copy = row.copy()
                    row_copy['embedding'] = embedding if embedding else "[]"
                    row_copy[self._configuration.embedColumn] = chunk
                    writer.writerow(row_copy)
                    self.row_count += 1




    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

    def init_openai_client(self):
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)

    def get_embedding(self, text, model = 'text-embedding-3-small'):
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
            out_table_name = f"app-embed-lancedb.csv"
        else:
            out_table_name = f"{out_table_name}.csv"

        return self.create_out_table_definition(out_table_name)


if __name__ == "__main__":
    try:
        comp = Component()
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
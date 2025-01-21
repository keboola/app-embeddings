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
            raise logging.info(f"Error occurred during embedding process: {str(e)}")

    def _process_rows_csv(self, reader):

        output_table = self._get_output_table()
        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
            fieldnames = reader.fieldnames + ['embedding']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            self.row_count = 0
            if self._configuration.chunkingEnabled:
                chunk = []
                for row in reader:
                    self.row_count += 1
                    text = row[self._configuration.embedColumn]
                    chunk.append(text)

                    if len(chunk) == self._configuration.chunkSize:
                        self._process_chunk(chunk, writer, row)
                        chunk = []

                if chunk:
                    self._process_chunk(chunk, writer, row)
            else:
                for row in reader:
                    self.row_count += 1
                    text = row[self._configuration.embedColumn]
                    embedding = self.get_embedding([text])[0]
                    row['embedding'] = embedding if embedding else "[]"
                    writer.writerow(row)

    def _process_chunk(self, chunk, writer, row_template):
        embeddings = self.get_embedding(chunk)

        for i, embedding in enumerate(embeddings):
            row = row_template.copy()
            row['embedding'] = embedding if embedding else "[]"
            writer.writerow(row)

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

    def init_openai_client(self):
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)

    def get_embedding(self, texts, model=None):
        if not texts or not isinstance(texts, list):
            return []
        texts = [text.replace("\n", " ") for text in texts if isinstance(text, str) and text.strip()]
        model = model or self._configuration.model
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    def _get_input_table(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        if len(self.get_input_tables_definitions()) > 1:
            raise logging.info("Only one input table is supported")
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

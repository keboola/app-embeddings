import csv
import logging
import os
import shutil
import zipfile
import lancedb
import json
import nltk

import pyarrow as pa
import pandas as pd

from typing import Iterator
from nltk.tokenize import sent_tokenize, word_tokenize


from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException
from configuration import Configuration

from openai import OpenAI
class TextChunker:
    @staticmethod
    def chunk_by_sentences(text: str, sentences_per_chunk: int = 3) -> Iterator[str]:
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            sentences = sent_tokenize(text)
            
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():
                yield chunk

    @staticmethod
    def chunk_by_words(text: str, words_per_chunk: int = 100, overlap: int = 20) -> Iterator[str]:
        words = word_tokenize(text)
        for i in range(0, len(words), words_per_chunk - overlap):
            chunk = ' '.join(words[i:i + words_per_chunk])
            if chunk.strip():
                yield chunk

    @staticmethod
    def chunk_by_chars(text: str, chars_per_chunk: int = 500) -> Iterator[str]:
        for i in range(0, len(text), chars_per_chunk):
            chunk = text[i:i + chars_per_chunk].strip()
            if chunk:
                yield chunk

    @staticmethod
    def no_chunking(text: str) -> Iterator[str]:
        if text.strip():
            yield text.strip()

class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None
        self.client = None
        self.chunker = None

    def run(self):
        self.init_configuration()
        self.init_client()
        self.init_chunker()
        
        input_table = self._get_input_table()
        with open(input_table.full_path, 'r', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file)
            if self._configuration.outputFormat == 'json':
                corpus_data = self._process_json_format(reader)
                output_file = os.path.join(self.tables_out_path, 'corpus.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(corpus_data, f, indent=2)
            else:
                output_table = self._get_output_table()
                self._process_csv_format(reader, output_table)

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

    def init_client(self):
        self.client = OpenAI(api_key=self._configuration.pswd_apiKey)

    def init_chunker(self):
        chunking_methods = {
            'none': TextChunker.no_chunking,
            'sentences': lambda text: TextChunker.chunk_by_sentences(text, self._configuration.chunk_size),
            'words': lambda text: TextChunker.chunk_by_words(text, self._configuration.chunk_size),
            'characters': lambda text: TextChunker.chunk_by_chars(text, self._configuration.chunk_size)
        }
        self.chunker = chunking_methods.get(self._configuration.chunk_method, TextChunker.no_chunking)

    def _process_json_format(self, reader):
        corpus_data = {
            "id": "corpus_001",
            "metadata": {
                "title": "Document Embeddings",
                "description": "Collection of documents with embeddings",
                "creationDate": self._get_current_date(),
                "lastModifiedDate": self._get_current_date(),
                "contributors": [],
                "version": "1.0",
                "dataSource": "Keboola Connection",
                "chunkingMethod": self._configuration.chunk_method,
                "chunkSize": self._configuration.chunk_size
            },
            "accessInformation": {
                "public": False,
                "authorizedUsers": []
            },
            "data": {
                "documents": [],
                "index": {
                    "keywords": [],
                    "documentIds": []
                }
            }
        }

        for doc_idx, row in enumerate(reader, 1):
            doc_id = f"doc_{doc_idx:03d}"
            text = row[self._configuration.embedColumn]
            document = {
                "id": doc_id,
                "title": row.get('title', f"Document {doc_idx}"),
                "nodes": []
            }

            for node_idx, chunk in enumerate(self.chunker(text), 1):
                embedding = self.get_embedding(chunk)
                node = {
                    "id": f"node_{doc_idx:03d}_{node_idx:03d}",
                    "content": chunk,
                    "embedding": embedding
                }
                document["nodes"].append(node)

            corpus_data["data"]["documents"].append(document)
            corpus_data["data"]["index"]["documentIds"].append(doc_id)

        return corpus_data

    def _process_csv_format(self, reader, output_table):
        with open(output_table.full_path, 'w', encoding='utf-8', newline='') as output_file:
            fieldnames = reader.fieldnames + ['chunk_id', 'chunk_text', 'embedding']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                text = row[self._configuration.embedColumn]
                for chunk_idx, chunk in enumerate(self.chunker(text)):
                    embedding = self.get_embedding(chunk)
                    row_copy = row.copy()
                    row_copy['chunk_id'] = chunk_idx
                    row_copy['chunk_text'] = chunk
                    row_copy['embedding'] = embedding
                    writer.writerow(row_copy)

    def get_embedding(self, text):
        try:
            response = self.client.embeddings.create(input=[text], model=self._configuration.model)
            return response.data[0].embedding
        except Exception as e:
            raise UserException(f"Error getting embedding: {str(e)}")

    def _get_input_table(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified!")
        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")
        return self.get_input_tables_definitions()[0]

    def _get_output_table(self):
        return self.create_out_table_definition(f"{self._configuration.destination.output_table_name}.csv")

    def _get_current_date(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

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
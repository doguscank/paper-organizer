# Paper Organizer

An LLM-based paper organizer desktop program.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Paper Organizer is a desktop application designed to help researchers and academics organize their papers using advanced language models. The application categorizes, generates embeddings, and stores papers in a vector database for easy retrieval and management.

## Features

- **Categorization**: Automatically categorize papers based on their content.
- **Embedding Generation**: Generate embeddings for papers to facilitate similarity searches.
- **Vector Database**: Store and manage paper embeddings in a vector database.

## Installation

To install the Paper Organizer, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/doguscank/paper-organizer.git
    cd paper-organizer
    ```

2. Install dependencies using Poetry:
    ```sh
    poetry install
    ```

3. Set up the environment variables:
    ```sh
    cp src/.env.template src/.env
    # Edit src/.env with your configuration
    ```

## Usage

To run the application, use the following command:

```sh
poetry run python main.py
```

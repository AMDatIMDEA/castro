#!/usr/bin/env python3
import os
import re
import sys
import argparse
import glob


def generate_header(software_name, version, copyright_year, organization, contact_email, author, author_email, license_url):
    def to_string(value):
        return value if isinstance(value, str) else ", ".join(value) if isinstance(value, (list, tuple)) else str(value)

    software_name = to_string(software_name)
    version = to_string(version)
    copyright_year = to_string(copyright_year)
    organization = to_string(organization)
    contact_email = to_string(contact_email)
    author = to_string(author)
    author_email = to_string(author_email)
    license_url = to_string(license_url)

    header = f"""
###############################################################################
#
#                                {software_name}   v{version}
#
#
#     Copyright (C) {copyright_year} {organization}
#     Contact: {contact_email}
#     Author: {author} ({author_email})
#
#     This file is part of {software_name}.
#
#     {software_name} is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation version 3 of the License.
#
#     {software_name} is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public Version 3 License
#     along with {software_name}.  If not, see <{license_url}>.
#
###############################################################################
"""
    return header.strip()

def add_header_to_file(file_path, header):
    with open(file_path, 'r+', encoding='utf-8') as file:
        content = file.read()
        if header.strip() in content:
            print(f"Header already present in: {file_path}")
            return
        file.seek(0)
        file.write(header + '\n' + content)
        print(f"Header added to: {file_path}")


def add_or_replace_header(file_path, header):
    header_start = "###############################################################################"
    header_pattern = re.compile(
        rf"^{header_start}.*?{header_start}",
        re.DOTALL | re.MULTILINE
    )

    with open(file_path, 'r+', encoding='utf-8') as file:
        content = file.read()

        if header_pattern.search(content):
            content = header_pattern.sub(header, content)
            print(f"Header replaced in: {file_path}")
        else:
            content = header + '\n\n' + content
            print(f"Header added to: {file_path}")

        file.seek(0)
        file.write(content)
        file.truncate()


def add_header_to_all_files(src_dir, header):
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                add_or_replace_header(file_path, header)


def add_header_to_specific_files(file_paths, header):
    for file_path in file_paths:
        if os.path.exists(file_path) and file_path.endswith(".py"):  # Process only Python files
            add_or_replace_header(file_path, header)
        else:
            print(f"Skipping non-Python file or invalid file path: {file_path}")


def main():
    # Define the header details
    software_name = "CASTRO"
    version = "1.1.2"
    copyright_year = "2025"
    organization = "Fundación IMDEA Materiales (IMDEA Materials Institute), Getafe, Madrid, Spain"
    contact_email = "christina.schenk@imdea.org"
    author = "Christina Schenk"
    author_email = "christina.schenk@imdea.org"
    license_url = "http://www.gnu.org/licenses/"

    header = generate_header(
        software_name,
        version,
        copyright_year,
        organization,
        contact_email,
        author,
        author_email,
        license_url
    )

    # Specify the src directory
    src_dir = "src"
    add_header_to_all_files(src_dir, header)
    file_paths = glob.glob("examples/*.py", recursive=True)
    add_header_to_specific_files(file_paths, header)
    add_header_to_all_files(src_dir, header)


if __name__ == "__main__":
    # Run the main function without args for testing purposes
    main()

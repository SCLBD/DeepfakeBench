#!/bin/bash

DIR="."
PASSWORD="123456" # Please replace with the actual zip extraction password (123456 by default)

# Function to handle errors
handle_error() {
  echo "An error occurred during the script execution."
  exit 1
}

# Validate inputs
if [[ ! -d "$DIR" ]]; then
  echo "Invalid directory: $DIR"
  exit 1
fi

# Iterate over all .zip files in the current directory and extract them
for file in "$DIR"/*.zip; do
  if [[ ! -f "$file" ]]; then
    echo "Invalid zip file: $file"
    continue
  fi

  echo "Unzipping $file"
  if ! unzip -P "$PASSWORD" "$file"; then
    echo "Failed to unzip $file"
    continue
  fi
done

# Delete the __MACOSX folder if it exists
if [[ -d "__MACOSX" ]]; then
  rm -rf __MACOSX
fi

echo "Dataset operation completed successfully!"
exit 0

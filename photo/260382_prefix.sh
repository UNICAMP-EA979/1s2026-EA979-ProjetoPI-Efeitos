
#!/bin/bash

# Usage: ./prefix_files.sh /path/to/folder 123456

DIR="$1"
PREFIX="260382"

if [ -z "$DIR" ] || [ -z "$PREFIX" ]; then
  echo "Usage: $0 /path/to/folder 6digitprefix"
  exit 1
fi

if [ ! -d "$DIR" ]; then
  echo "Error: Directory does not exist"
  exit 1
fi

for file in "$DIR"/*; do
  [ -f "$file" ] || continue

  filename=$(basename "$file")

  # Skip if already prefixed
  if [[ "$filename" == "$PREFIX"_* ]]; then
    echo "Skipping already prefixed: $filename"
    continue
  fi

  newname="${PREFIX}_${filename}"

  mv "$file" "$DIR/$newname"
  echo "Renamed: $filename -> $newname"
done

echo "Done."

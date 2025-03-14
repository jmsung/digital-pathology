#!/bin/bash

for ext in png pth; do
    for f in *Ext."$ext"; do
        if [[ "$f" != *_0."$ext" ]]; then
            newname="${f%.$ext}_0.$ext"
            echo "Renaming '$f' to '$newname'"
            mv "$f" "$newname"
        fi
    done
done

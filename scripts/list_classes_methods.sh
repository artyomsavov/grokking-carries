#!/usr/bin/env bash
# scripts/list_classes_methods.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src/grokking_carries"

if [ ! -d "$SRC_DIR" ]; then
    echo "Ошибка: директория $SRC_DIR не найдена. Запускайте скрипт из корня проекта."
    exit 1
fi

find "$SRC_DIR" -name "*.py" -print0 | sort -z | while IFS= read -r -d '' file; do
    rel="${file#$PROJECT_ROOT/}"
    echo ""
    echo "$rel"
    python3 -c "
import ast, sys
with open('$file', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read(), filename='$file')
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        print(f'  {node.name}')
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # пропустить магические методы, если нужно — убрать фильтр
                # if not item.name.startswith('__'):
                print(f'     {item.name}')
"
done


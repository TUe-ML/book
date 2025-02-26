#!/bin/sh


cd ./images/classification


FILES="*.tex"

for FILE in $FILES
do
  echo "Processing $FILE file..."
  pdflatex -synctex=1 -interaction=batchmode -shell-escape $FILE
done

cd ../neuralnets

FILES="*.tex"

for FILE in $FILES
do
  echo "Processing $FILE file..."
  pdflatex -synctex=1 -interaction=batchmode -shell-escape $FILE
done

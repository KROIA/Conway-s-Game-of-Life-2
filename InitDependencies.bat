@echo off
echo Downloading dependencies...

git submodule update --init --recursive

echo Downloading done

echo Build dependencies...
cd dependencies/SFML_EditorWidget
start build.bat
pause
#!/bin/bash
cd /home/sig07/pronostico_rutas
git init

# Cargar las variables de entorno desde un archivo no versionado
source /home/sig07/pronostico_rutas/env.sh

export PATH="$PATH:/home/sig07/.local/bin"
python3 /home/sig07/pronostico_rutas/gitscript.py


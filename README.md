# Pipeline de Veeduría Electoral — Actas E-14 Colombia 2026

Herramienta de veeduría ciudadana para la descarga masiva y análisis automatizado de las actas de escrutinio E-14 de las Elecciones Presidenciales de Colombia 2026. El propósito es habilitar la detección de alteraciones manuscritas mediante modelos de visión artificial (enmendaduras, sobrescrituras e inconsistencias aritméticas).

---

## Contexto técnico

El portal oficial de divulgación (`divulgacione14presidente.registraduria.gov.co`) es una Single Page Application compilada en Angular 19 que consume datos via AWS AppSync (GraphQL) y recibe actualizaciones en tiempo real por WebSocket. Los PDFs se sirven desde una arquitectura de almacenamiento denominada internamente *Temis*, protegida por Akamai CDN con restricción geográfica a IPs colombianas.

Las actas no se exponen con nombres secuenciales. Cada archivo PDF tiene un nombre basado en un hash alfanumérico único asignado en el momento de la transmisión. El script reconstruye las URLs a partir de un archivo de índice plano (`e14_transmission_index_2026.json`) que el cliente Angular carga en memoria al iniciar, y que fue interceptado para construir este pipeline.

---

## Estructura del repositorio

```
.
├── tally-sheets-download.py        Script principal de descarga
├── e14_transmission_index_2026.json  Indice de transmision (ignorado por git, ~63 MB)
├── .env.example                    Plantilla de configuracion
├── .env                            Configuracion local (ignorado por git)
├── .gitignore
└── README.md
```

---

## Prerrequisitos

### Software

- **uv** >= 0.4 — gestor de entornos y dependencias de Python. El shebang del script (`#!/usr/bin/env -S uv run`) instala automáticamente `httpx`, `tqdm` y `python-dotenv` en un entorno aislado sin necesidad de instalación manual.

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Python** >= 3.9 (gestionado por uv, no requiere instalación separada).

### Requisito de red: IP colombiana

El CDN de Akamai que protege el portal aplica geo-restricción y rechaza conexiones fuera de Colombia con un `HTTP/2 INTERNAL_ERROR` inmediato o timeout en HTTP/1.1. El script **debe ejecutarse desde una IP colombiana**:

- VPS o instancia cloud con punto de presencia en Colombia (proveedores como ETB, Claro, Telmex Colombia).
- VPN con nodo de salida en Colombia.

---

## Configuración

Copie la plantilla y ajuste los valores según su entorno:

```bash
cp .env.example .env
```

| Variable        | Descripción                                                  | Default                              |
|-----------------|--------------------------------------------------------------|--------------------------------------|
| `OUTPUT_DIR`    | Directorio de destino para los PDFs                          | `./actas`                            |
| `INDEX_FILE`    | Ruta al índice de transmisión JSON                           | `e14_transmission_index_2026.json`   |
| `CONCURRENCY`   | Número de descargas paralelas                                | `20`                                 |
| `RETRIES`       | Reintentos por archivo ante fallo de red                     | `3`                                  |
| `ERRORS_FILE`   | Ruta del log CSV con registros fallidos                      | `errors.csv`                         |
| `STATUS_FILTER` | Filtrar por estado de transmisión (ver tabla de estados)     | *(sin filtro)*                       |

Los parámetros de línea de comandos tienen prioridad sobre los valores del `.env`.

### Estados de transmisión

| Código | Significado                                      | Registros |
|--------|--------------------------------------------------|-----------|
| `11`   | Acta transmitida y confirmada (conjunto principal) | 119,856  |
| `3`    | Transmisión parcial o pendiente                  | 345       |

Para la descarga de producción se recomienda `STATUS_FILTER=11`.

---

## Uso

### Descarga de producción (desde IP colombiana)

```bash
uv run tally-sheets-download.py --status 11
```

Con todos los parámetros explícitos:

```bash
uv run tally-sheets-download.py \
  --index e14_transmission_index_2026.json \
  --out ./actas \
  --concurrency 20 \
  --retries 3 \
  --status 11 \
  --errors errors.csv
```

### Prueba con un subconjunto

```bash
uv run tally-sheets-download.py --status 11 --limit 10 --out actas_test/
```

### Reanudación automática

Si la descarga se interrumpe, basta con volver a ejecutar el mismo comando. El script detecta los archivos ya existentes con tamaño mayor a cero y los omite.

### Reintentar solo los fallidos

```bash
# errors.csv contiene las URLs que fallaron; se puede reprocesar manualmente
# o incorporar al pipeline de reintentos en futuras iteraciones del script.
cat errors.csv
```

---

## Estructura del índice JSON

El archivo `e14_transmission_index_2026.json` tiene la siguiente estructura:

```json
{
  "data": {
    "status11": {
      "nodes": [
        {
          "idTransmissionCode": "3497118",
          "numberStand": "027",
          "expectedName": "bf247ca2...dc41.pdf",
          "idTransmissionCodeStatus": 11,
          "idCorporationCode": "001",
          "idStand": "000011815",
          "standCode": "00",
          "idZoneCode": "00",
          "idDepartmentCode": "15",
          "municipalityCode": "118"
        }
      ]
    },
    "status3": { "nodes": [ ... ] }
  }
}
```

---

## Convención de nombres de los archivos descargados

Cada PDF se guarda con el siguiente formato, pensado para facilitar la ingestión en pipelines de visión artificial sin requerir metadatos adicionales:

```
E14_PRE_{depto}_{municipio}_{zona:3}_{puesto:2}_{mesa:3}.pdf
```

Ejemplo:
```
E14_PRE_15_118_000_00_027.pdf
```

| Segmento    | Fuente en el índice      | Padding |
|-------------|--------------------------|---------|
| `depto`     | `idDepartmentCode`       | Ninguno |
| `municipio` | `municipalityCode`       | Ninguno |
| `zona`      | `idZoneCode`             | 3 dígitos (zfill) |
| `puesto`    | `standCode`              | 2 dígitos (zfill) |
| `mesa`      | `numberStand`            | 3 dígitos (zfill) |

---

## URL de descarga

La URL se construye siguiendo la arquitectura Temis del portal:

```
https://divulgacione14presidente.registraduria.gov.co/assets/temis/pdf/
  {idDepartmentCode}/
  {municipalityCode}/
  {idZoneCode:3}/
  {standCode:2}/
  {numberStand:3}/
  PRE/
  {expectedName}.pdf
```

---

## Consideraciones operativas

- **Volumen:** ~120,000 PDFs de 3 páginas cada uno. Estimar entre 300 MB y 1.5 GB de almacenamiento total dependiendo del tamaño promedio por acta.
- **Concurrencia:** El valor por defecto de 20 conexiones paralelas es conservador. Aumentar `CONCURRENCY` si el ancho de banda lo permite; reducir si el servidor devuelve errores 429 o timeouts frecuentes.
- **Backoff exponencial:** Ante un fallo de red, el script espera 2, 4 u 8 segundos entre reintentos según el número de intento.
- **Integridad:** Los archivos se escriben primero con extensión `.tmp` y se renombran al completarse, evitando PDFs corruptos si el proceso se interrumpe.
- **Memoria:** La carga del índice JSON (~63 MB en disco) puede ocupar 200-350 MB en memoria como objeto Python. Se recomienda un mínimo de 512 MB de RAM disponible.

---

## Aviso legal y ético

Este proyecto tiene fines exclusivos de **veeduría ciudadana y transparencia electoral**. Las actas E-14 son documentos públicos conforme al artículo 74 de la Constitución Política de Colombia y a los principios de publicidad y transparencia del Código Electoral. El acceso a los datos se realiza desde el portal oficial de la Registraduría Nacional del Estado Civil. No se almacenan datos personales ni se realizan peticiones que excedan el uso previsto de la infraestructura pública.

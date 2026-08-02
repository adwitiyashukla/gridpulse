{#
    Keep every model in the `main` schema of the DuckDB file rather than
    creating per-model schemas. With a single-file embedded warehouse the
    extra nesting buys nothing and complicates the read path for the app.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ default_schema }}
{%- endmacro %}

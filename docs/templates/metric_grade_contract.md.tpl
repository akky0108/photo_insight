# {{ title }}

## Overview
{{ overview }}

## Grade enum (SSOT)
{% for g in grade_enum -%}
- `{{ g }}`
{% endfor %}

## Score to grade mapping
| score | grade |
|---:|---|
{% for s, g in score_to_grade.items() -%}
| {{ s }} | `{{ g }}` |
{% endfor %}

## eval_status policy
{% for k, v in status_policy.items() -%}
- `{{ k }}`: {{ v }}
{% endfor %}

## Legacy grade aliases (normalization)
{% for k, v in legacy_grade_aliases.items() -%}
- `{{ k }}` -> `{{ v }}`
{% endfor %}

## Notes
{% for n in notes -%}
- {{ n }}
{% endfor %}

# {{ name }} Design & Contract

## 1. Overview

{{ overview }}

---

## 2. Output Contract

### 2.1 Output Fields

| Field | Type | Description |
|-------|------|-------------|
{% for f in fields -%}
| {{ f.name }} | {{ f.type }} | {{ f.desc }} |
{% endfor %}

---

## 3. Purpose

{{ purpose }}

---

## 4. Algorithm

{{ algorithm }}

---

## 5. Normalization Policy

{{ normalization }}

---

## 6. Fallback Policy

{{ fallback }}

---

## 7. Testing

{{ testing }}

---

## 8. Future Plans

{{ future }}

---

## 9. Design Philosophy

{{ philosophy }}

---

## 10. Change History

| Date | Change | Author |
|------|--------|--------|
{% for c in history -%}
| {{ c.date }} | {{ c.change }} | {{ c.author }} |
{% endfor %}

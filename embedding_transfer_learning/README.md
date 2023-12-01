{% for _ in cookiecutter.project_name %}={% endfor %}
{{ cookiecutter.project_name }}
{% for _ in cookiecutter.project_name %}={% endfor %}

{{ cookiecutter.project_short_description }}

_______
Author:
{% for _ in cookiecutter.author %}-{% endfor %}
{{ cookiecutter.author }}, {{ cookiecutter.email }}


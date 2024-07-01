
auto-generate a django project:
```bash
django-admin startproject <project name>
```

Note: when putting this on a server, you slhould put this in some directory outside of the document root

initial directory: 
```
<sitename>/
	manage.py
	<sitename>/
		__init__.py
		settings.py
		urls.py
		asgi.py
		wsgi.py

```

- outer directory: doesn't matter to django, we can rename it to whatever we want
- manage.py - a command-line utility that lets you interact with django in different ways
- ``<sitename> / <sitename>/`` - actual python package for the project
- `<sitename> / __init__.py` - tells python to treat mysite as a package
- `<sitename> / setti ngs.py` - settings and configuration
- `<sitename> / urls.py` - url declarations, think of it as a table of contents
- `<sitename> / asgi.py` - entry-point for ASGI-compatible web servers
- `<sitename> / wsgi.py` - entry-point for WSGI-compatible servers

**Run the develoipment server**
`python manage.py runserver`

**app** - a python package that foillows a certain convention. This is a web application that does something (e.g. blog system, database of public records, etc.)
**project** - a collection of configuration and apps for a particular website

**Create an app**
`python manage.py startapp <app name>`

the app will be laid out like this:

```
polls/
	__init__.py
	admin.py
	apps.py
	migrations/
		__init__.py
	models.py
	tests.py
	views.py
```

In the tutorial, wrote an index function in views.py

To be able to call the view, map it to a url. Do this by making a URLconf.

- create a file called urls.py in the app's directory
- put this in urls.py:
```
from django.urls import path
 
from . import views
 
 urlpatterns = [
         path("", views.index, name='index'),
        ]

```

Now , we want to point the root URLconf at polls.urls moodule

put this inside the root urls.py:
```python
 16 from django.contrib import admin
 17 from django.urls import include, path
 18 
 19 urlpatterns = [
 20     path('admin/', admin.site.urls),
 21     path("polls/", include("polls.urls")),
 22 ]
```

When django sees `include()`, it'll chop off whatever part of lthe url matched up to that point and send the remaining string to `polls.urls`. So, if the url is `page/another_page/polls/other/stuff/`, it will send `other/stuff/` to the URLconf in polls.urls for other processing. 



# Resources
https://docs.djangoproject.com/en/4.1/intro/tutorial01/
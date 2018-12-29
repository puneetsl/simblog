![](http://i.imgur.com/uZjkPI5.png)

-----

`Version: 0.01`

### Why Simblog

After searching a lot, I found no good `hosted`, `free`, `markdown-based` blogging platform. [Ghost](https://ghost.org) is the closest to programmer friendly blogging platform but the hosted solution is [too expensive](https://ghost.org/pricing/) for an enthusiast blogger.
Medium is also a good contender but there is not good code snippet support, creating gist for everything is very frustrating at times.
I found that setting up Jekyll on Github pages is the closest to the solution I was trying to find. The only problem is the overhead of setting it up is too much. Also, changing theme is a massive project in itself sometimes.

So, `Simblog` aims to make this process as easy as possible. Its not only code but also a tutorial on how you can blog as easy as possible, using the tools you love.

### Setup


1. **<a href="https://github.com/puneetsl/simblog/fork">Fork</a> Simblog**
2. Go to project settings and `Rename` your project to a name of your choise.(most people would chose `blog`)
![](http://i.imgur.com/h8rToJV.png)

3. Next, in settings select option to publish Github pages
![](http://i.imgur.com/vzGaf5Z.png)
4. Clone your project locally using
```sh
git clone git@github.com:<username>/<repo>.git
```
5. At this point you can use your favorite editor to start your blog. I will suggest using [Atom](https://atom.io/) or [VS Code](https://code.visualstudio.com/) for best blogging experience.
6. Before we beging setting up our editor for blogging make sue you have `python` and `pip` installed in your system and run the following command
```
pip install -r _scripts/requirements.txt
```
7. Since I prefer atom text editor for this blogging purpose, I would explain how to set it up.
    * if you know what is [apm](https://github.com/atom/apm) and have installed it you can run the following command
    ```sh
    apm install --packages-file atom-package-list.txt
    ```
    * if not, press `CTRL+SHIFT+P` in atom and type `install packages and themes` and install following packages one by one
      ```
      busy-signal
      date
      intentions
      language-markdown
      linter
      linter-markdown
      linter-ui-default
      markdown-image-paste
      markdown-pdf
      markdown-preview-enhanced
      markdown-toc
      markdown-writer
      platformio-ide-terminal
      script
      tidy-markdown
      tool-bar
      tool-bar-markdown-writer
      ```
    * Restart Atom
    * At this point if you open a markdown file it should look like this
    ![](http://i.imgur.com/AgaZEgD.png)
8. We will use python to help us, and to auto run `python` in atom we will use script plugin. you can see the shortcut like this:
![](http://i.imgur.com/CvvDnWe.png)
9. **That's it, your setup is done**

### New post
![](http://i.imgur.com/Fly5tjt.png)
Open `new_post.py` and press `CTRL+SHIFT+B`(or run this script with the shortcut we saw)
You will see this UI
![](http://i.imgur.com/29rHp11.png)
Now you can see a file generated in `_posts` folder
![](http://i.imgur.com/hMXVUWq.png)

And now your can write your post
![](http://i.imgur.com/UJFatoW.png)

### Publish
Run `_scripts/publish.py` and thats it, you will see this message
![](http://i.imgur.com/SGNEXlB.png)

-----
#### View the blog post at: http://www.puneetsingh.me/simblog/Test-blog-post.html
-----

### Todo
* [ ] Introduce themes
* [ ] introduce `setup.py` to set it up
* [ ] Make a custom electron app with all these features

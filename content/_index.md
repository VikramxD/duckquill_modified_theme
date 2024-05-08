+++
+++

{% crt() %}
```
      _          _          _          _          _
    >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,
      (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/
jgs~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~
```
{% end %}

# Duckquill

Duckquill is a modern, pretty, and clean (and opinionated) [Zola](https://www.getzola.org) theme that has the purpose of greatly simplifying the process of rolling up your blog. It aims to provide all the needed options for comfortable writing, keeping the balance of it being simple.

Edit a bit of metadata and tweak some of the included graphics and have a blog up in minutes!

- Pretty, yet lightweight. No JavaScript is used (except for comments, heh).
- Colors are tinted with the user-selected primary color for a pleasent look.
- ~50kB in size; take that, 12MB Medium!
- Uses system fonts for blazingly fast loading.
- Mobile friendly, with a proper dark variant.
- Proper favicon for modern browsers and Apple device icons.
- Twitter, Mastodon and other social media meta cards for easy sharing. Try [Share Preview](https://apps.gnome.org/SharePreview/) to test.
- [Mastodon-powered comments](https://carlschwan.eu/2020/12/29/adding-comments-to-your-static-blog-with-mastodon/); comment using compatible ActivityPub server by replying to a Mastodon post.

> Duckquill is made based on needs of [my website](https://daudix.codeberg.page), if you need some feature/configuration that doesn't exist feel free to open an issue or better yet, pull request!

## Installation

First, download this theme to your `themes` directory:

```sh
git clone https://codeberg.org/daudix/duckquill.git themes/duckquill
```

...or add as submodule for easy updating (recommended if you already have git setup on site):

```sh
git submodule init
git submodule add https://codeberg.org/daudix/duckquill.git themes/duckquill
```

It is highly recommended to switch from the `main` branch to the lates release:

```sh
cd themes/duckquill
git checkout tags/v1.1.0
```

To update the submodule, simply switch to a new tag:

> **Note:** Check the changelog for all versions that came after the one you are using, there may be some breaking changes that need manual involvement.

```sh
git submodule update --remote --merge
git tag --list
git checkout tags/v1.1.0
```

Then, enable it in your `config.toml`:

```toml
theme = "duckquill"
```

## Options

Duckquill offers some configuration options to make it fit you better; most options have pretty descriptive comments, so it should be easy to understand what they do.

### Custom stylesheets

You can add your own or override existing styles by creating custom stylesheet and adding it in the `config.toml`:

```toml
[extra]
stylesheets = [
  "YOUR_STYLE.css"
]
```

Additional stylesheets; expects it to be in the `static` directory. If you are using Sass it will be compiled there anyway.

If for some reason overridden class are not respected, try using `!important`. You can import styles from Duckquill using:

```scss
@use "../themes/duckquill/sass/NEEDED_FILE.scss";
```

### Primary color

Duckquill respects chosen primary color everywhere, simply change the primary color in `config.toml`:

```toml
[extra]
primary_color = "COLOR_CODE"
primary_color_alpha = "COLOR_CODE"
```

## Test pages

- [Demo page](@/demo/index.md)
- [Cake Party!](@/demo/page.md)
- [ActivityPub/​Fediverse comments demo](@/demo/comments.md)
- [Code block demo (needs Zola v0.18.0 and higher)](@/demo/code.md)

## Sites that use Duckquill

- [agustinramirodiaz.github.io](https://agustinramirodiaz.github.io)
- [alavi.me](https://alavi.me)
- [bano.dev](https://bano.dev)
- [blog.pansi21.xyz](https://blog.pansi21.xyz)
- [daudix.codeberg.page](https://daudix.codeberg.page) (obviously)
- [daveparr.info](https://www.daveparr.info)
- [digital-horror.com](https://digital-horror.com)
- [enriquekesslerm.com](https://enriquekesslerm.com)
- [jzbor.de](https://jzbor.de)
- [licu.dev](https://licu.dev)
- [luciengheerbrant.com](https://luciengheerbrant.com)
- [lukoktonos.com](http://www.lukoktonos.com)
- [malloc.garden](https://malloc.garden)
- [mourelask.xyz](https://mourelask.xyz)
- [nbenedek.me](https://nbenedek.me)
- [notaplace.com](https://notaplace.com)
- [pyter.at](https://pyter.at)
- [rbd.gg](https://www.rbd.gg)
- [rerere.unlogic.co.uk](https://rerere.unlogic.co.uk)
- [sorg.codeberg.page](https://sorg.codeberg.page)
- [sungsphinx.codeberg.page](https://sungsphinx.codeberg.page)
- [treeniks.github.io](https://treeniks.github.io)
- Yours?

## Sites that mention Duckquill in credits

- [andreatitolo.com](https://www.andreatitolo.com/credits)
- [aplos.gxbs.me](https://aplos.gxbs.me)
- [oomfie.town](https://oomfie.town/credits)

## Credits

- [Quill image used in the metadata card](https://commons.wikimedia.org/wiki/File:3quills.jpg)

## Thanks to ♥

- [Jakub Steiner](https://jimmac.eu) for the [OS Component Website](https://jimmac.github.io/os-component-website), which served as a starting point and inspiration.
- [Carl Schwan](https://carlschwan.eu) for the [Mastodon-powered Comments](https://carlschwan.eu/2020/12/29/adding-comments-to-your-static-blog-with-mastodon/).
- [Jonathan Neal](https://jonneal.dev) for the [normalize.css](https://csstools.github.io/normalize.css/).
- Everyone who supported me and said good stuff <3

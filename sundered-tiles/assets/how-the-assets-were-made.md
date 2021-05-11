# How the assets were made

## Initial tiles

We started with the vector assets from [this art pack by Kenney.nl](https://opengameart.org/content/sokoban-100-tiles).

Then I selected some tiles and recoloured them in inkscape. Then we exported to GIMP, and found that there was some internal transparency shere the vector edges didn't line up exactly. So we put an underlay of the main colour of each tile, under each tile and put that in the spritesheet. When doing that, we made sure to crop off the outermost semi-transparent layer, leaving an apron of empty pixels around the edge of each tile.

After that we needed a set of special versions of the coloured tiles, that indicate where all the other tiles of that colour are. After some reflection a compass rose seemed appropriate, so I rendered a semi-transparent one using the GIMP Gfig filter.

I also wanted a tile that indicates that the level is over, but there is another one, and a different tile that indicates that you've found the thing you were looking for. After some thought I hit upon the idea of a treasure map for the first. I then recalled how Hyperrogue had a level where compasses pointed to treasures that are literal Xs as in "X marks the spot" on a treasure map.

I then looked for a treasure map or scroll icon and found [this collection](https://opengameart.org/content/cc0-document-icons) which contains a treasure map icon from [this page](https://www.deviantart.com/7soul1/art/129892453) where it is described as "Public Domain This work is free of known copyright restrictions".

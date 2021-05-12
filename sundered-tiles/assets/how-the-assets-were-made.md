# How the assets were made

## Initial tiles

We started with the vector assets from [this art pack by Kenney.nl](https://opengameart.org/content/sokoban-100-tiles).

Then I selected some tiles and recoloured them in inkscape. Then we exported to GIMP, and found that there was some internal transparency shere the vector edges didn't line up exactly. So we put an underlay of the main colour of each tile, under each tile and put that in the spritesheet. When doing that, we made sure to crop off the outermost semi-transparent layer, leaving an apron of empty pixels around the edge of each tile.

After that we needed a set of special versions of the coloured tiles, that indicate where all the other tiles of that colour are. After some reflection a compass rose seemed appropriate, so I rendered a semi-transparent one using the GIMP Gfig filter.

I also wanted a tile that indicates that the level is over, but there is another one, and a different tile that indicates that you've found the thing you were looking for. After some thought I hit upon the idea of a treasure map for the first. I then recalled how Hyperrogue had a level where compasses pointed to treasures that are literal Xs as in "X marks the spot" on a treasure map.

I then looked for a treasure map or scroll icon and found [this collection](https://opengameart.org/content/cc0-document-icons) which contains a treasure map icon from [this page](https://www.deviantart.com/7soul1/art/129892453) where it is described as "Public Domain This work is free of known copyright restrictions".

The map was originally 32 * 31. I doubled the size with no interpolation to get a 64 by 62 image. I then selected just the pixels of the X, by using the select by colour tool to subtract the base map colour from a rectangle selection. Then I used the Hue-Chroma tool and set the Hue, Chroma, and Lightness to -40, 50, and -40 respectively, in order to make it red. I then just stuck that map in the center of the blank tile. 

Thne to make the standalone X I selected the now red X again. It was 14 by 16, so I again scaled it up withut interpolation, this time to 56 by 64 so it is the same relative size as the map, hoping that the lack of interpolation would make it more recognizable as te exact X from the map. Again, this was placed in the center of the blank tile.

I then decided that the X was too small to see on the map, so I clumsily doubled its size and moved some of the detailing to make it fit.



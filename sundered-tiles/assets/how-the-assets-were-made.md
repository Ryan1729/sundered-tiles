# How the assets were made

## Initial tiles

We started with the vector assets from [this art pack by Kenney.nl](https://opengameart.org/content/sokoban-100-tiles).

Then I selected some tiles and recoloured them in inkscape. Then we exported to GIMP, and found that there was some internal transparency shere the vector edges didn't line up exactly. So we put an underlay of the main colour of each tile, under each tile and put that in the spritesheet. When doing that, we made sure to crop off the outermost semi-transparent layer, leaving an apron of empty pixels around the edge of each tile.

After that we needed a set of special versions of the coloured tiles, that indicate where all the other tiles of that colour are. After some reflection a compass rose seemed appropriate, so I rendered a semi-transparent one using the GIMP Gfig filter.

I also wanted a tile that indicates that the level is over, but there is another one, and a different tile that indicates that you've found the thing you were looking for. After some thought I hit upon the idea of a treasure map for the first. I then recalled how Hyperrogue had a level where compasses pointed to treasures that are literal Xs as in "X marks the spot" on a treasure map.

I then looked for a treasure map or scroll icon and found [this collection](https://opengameart.org/content/cc0-document-icons) which contains a treasure map icon from [this page](https://www.deviantart.com/7soul1/art/129892453) where it is described as "Public Domain This work is free of known copyright restrictions".

I selected just the pixels of the X, by using the select by colour tool to subtract the base map colour from a rectangle selection. Then I used the Hue-Chroma tool and set the Hue, Chroma, and Lightness to -40, 50, and -40 respectively, in order to make it red.

Then to make the standalone X I selected the now red X again cut and pasted it.

I then needed to decide on the scaling. I found that anything smaller than the while tile made the map hard to read. So I scaled the map up with no interpolation to the largest multiple that would fit.

As for the lone X, the scaled up version of the X from the map didn't look good from a distance. So I tried scaling it various ways and eventually ended up redrawing it and scaling it up. This process involved scaling up the X a little and then cutting away the pointy edges that resulted from the corners of pixels, then scaling up some more.

Undertiled had a little information symbol for the hints, so I drew one of those myself. Specifically, I copied a blank grey tile and then selected a 96 by 96 circle in the center, then filled it in with blue (#3352e1). I decided that 1/3 of the way doen the tile would be a good place for the dot of the "i" so I changed the grid to be 64 by 43 so things would snap to the right spot. I then selected a 16 by 16 circle and placed it at that spot and filled it in with white. Then I changed the grid to 64 by 8 so I could snap a rectangle to a point that is close to the white circle. I thne fselected a 12 by 48 rectangle and placed it such that the top of the rectangle was at the point just below the bottomost grid square that overlapped the circle. I then filled that rectangle in white as well. But then I decided that I wanted rounded courners so I un-did the fill, checked that checkbox, left the radius value at 10, and re-filled that selection.




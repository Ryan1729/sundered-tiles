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

I decided that I wanted a pictoral representation of the hints, alongside, (or possibly even replacing) the text version. As part of that, we need a way to note the differnce between the edge of the grid and a "normal" empty space. We could treat them the same but adding ambiguity there doesn't seem like it makes for a better puzzle. So, I set the grid to 64 by 4 to make it easy to select a 120 by 4 line at the bottom of a blank, (all 0 alpha) sprite. I then added an arrow by drawing one with the select tool and the bucket tool with the grid at 4 by 4, but I purposely sometimes selected only "half-aligned" to the grid. After trying it out in the game, I remebered that we were clipping off the edges of the tile graphics, so I clipped a bit off the arrow stem and moved the line away from the edge so the line was visible.

For the pictoral hints, I needed a tile to represent the unknown, that is, what the hint is not telling you about. A question mark seemed appropriate. So I reduced down the grid again and placed a 16 by 16 circular ellipse selection horizontanlly centered and placed vertically such that the bottom edge was in the middle of the edge line from the previous tile, and then filled it it with the bucket tool. I fugured a somewhat symmetrical question mark made out of similar 16 by 16 circles would be good, so I set the grid to 32 by 4 and placed the leftmost and rightmost circles by snapping to the grid points at a y of 40. I then placed the top circle vertically opposite the bottom dot, (the same distance from the top, that one was from the bottom.) I then completed the diamond made by the last three placed points. Then I filled in the rest of the question mark by adding many layers with copies of one of the previous 16 by 16 circles on it, then moving them around until I felt comfortable merging the layers down, extending the tail down as far as seemed appropriate. But then that looked weird so I ended up using the path tool with a control point on the left and right dot, one on the vertally middle dot, and another slightly further towards the tail. I then used the stroke path option to draw the path. Unfortunately, the path tool doesn't seem to show numbers for control point positions, so I can't give you any! What I cansay is that I tried to have the path pass through the centers of the previously placed circles. I then decided that the bottom dot should be larger, so I selected the same 16 by 16 area and made a 20 by 20 circle selection on the same spot and filled it in.




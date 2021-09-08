* actual "minimum distance to nearest blah tile"
    * use DistanceIntel!
    * This seems very slightly more powerful than plain distance. I think that's probably okay.
    * I think passing adding a tile access clousre to `minimum_distance_between`, 0, and maximum tile::XY should work for this
        * The tile access closure would map the 0 to max tile::XY to the actual entire tile space.
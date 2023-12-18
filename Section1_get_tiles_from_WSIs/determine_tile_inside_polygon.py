import line_segments_intersect_or_not
import point_lies_inside_polygon_or_not

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Tile:
    def __init__(self, x, y, dx, dy):
        self.x = x#start point
        self.y = y#start point
        self.dx = dx#side length
        self.dy = dy#side length

# Returns true if the tile lies 
# inside the polygon[] with n vertices
# The last point in points should not be same as first point!!!!!!!
#Check if tile is totally inside polygon
def tile_is_inside_polygon(points, tile) -> bool:
    n = len(points)
    if n < 3:
        return False
    tile_points=[Point(tile.x,tile.y), Point(tile.x+tile.dx,tile.y), \
    Point(tile.x+tile.dx,tile.y+tile.dy), Point(tile.x,tile.y+tile.dy)]

    i=0
    while True:
        if i==n-1:
            next_point_index=0
        else:
            next_point_index=i+1
        
        if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[0],tile_points[1]):
            return False
        if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[1],tile_points[2]):
            return False
        if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[2],tile_points[3]):
            return False
        if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[3],tile_points[0]):
            return False

        i=next_point_index
        if i==0:
            break
    
    #return point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x,tile.y))
    # We could have case that tile.y is same as polygon's line segment's vertice's .y, then we have algorithm error in some cases. So we add extra 0.919(prime number)
    # (we didn't use 0.5 because if we downsample, line segment coordinates would be X.5)
    
    count=0
    if point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x+0.919,tile.y+0.919)):
        count+=1
    if point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x+1.919,tile.y+1.919)):
        count+=1
    if point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x+2.919,tile.y+2.919)):
        count+=1
    if count>=2:
        return True
    else:
        return False

def tile_CENTER_is_inside_polygon(points, tile) -> bool:
    n = len(points)
    if n < 3:
        return False
    # tile_points=[Point(tile.x,tile.y), Point(tile.x+tile.dx,tile.y), \
    # Point(tile.x+tile.dx,tile.y+tile.dy), Point(tile.x,tile.y+tile.dy)]

    # i=0
    # while True:
    #     if i==n-1:
    #         next_point_index=0
    #     else:
    #         next_point_index=i+1
        
    #     if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[0],tile_points[1]):
    #         return False
    #     if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[1],tile_points[2]):
    #         return False
    #     if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[2],tile_points[3]):
    #         return False
    #     if line_segments_intersect_or_not.doIntersect(points[i],points[next_point_index],tile_points[3],tile_points[0]):
    #         return False

    #     i=next_point_index
    #     if i==0:
    #         break
    
    #return point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x,tile.y))
    # We could have case that tile.y is same as polygon's line segment's vertice's .y, then we have algorithm error in some cases. So we add extra 0.919(prime number)
    # (we didn't use 0.5 because if we downsample, line segment coordinates would be X.5)
    
    count=0
    if point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x+0.919-2.0+tile.dx/2,tile.y+0.919-2.0+tile.dy/2)):
        count+=1
    if point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x+1.919-2.0+tile.dx/2,tile.y+1.919-2.0+tile.dy/2)):
        count+=1
    if point_lies_inside_polygon_or_not.is_inside_polygon(points, Point(tile.x+2.919-2.0+tile.dx/2,tile.y+2.919-2.0+tile.dy/2)):
        count+=1
    if count>=2:
        return True
    else:
        return False

if __name__ == '__main__':
    polygon=[ Point(0, 0), Point(0, 100), Point(100, 100), Point(20, 80), Point(20, 20), Point(120,0) ]
    tile1=Tile(50,50,10,10)
    print(tile_is_inside_polygon(polygon, tile1))

    tile2=Tile(5,70,5,5)
    print(tile_is_inside_polygon(polygon, tile2))

    tile3=Tile(-20,50,10,10)
    print(tile_is_inside_polygon(polygon, tile3))

    tile4=Tile(-5,-5,10,10)
    print(tile_is_inside_polygon(polygon, tile4))

    tile5=Tile(-5,60,40,40)
    print(tile_is_inside_polygon(polygon, tile5))

    tile6=Tile(-10,-10,200,200)
    print(tile_is_inside_polygon(polygon, tile6))

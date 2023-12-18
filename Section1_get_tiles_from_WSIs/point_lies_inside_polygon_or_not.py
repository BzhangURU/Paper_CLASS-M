# This code is modified version of https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
# This code is contributed by Vikas Chitturi, but is modified 

#python point_lies_inside_polygon_or_not.py

# A Python3 program to check if a given point 
# lies inside a given polygon
# Refer https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# for explanation of functions onSegment(),
# orientation() and doIntersect() 
 
import line_segments_intersect_or_not
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
# Define Infinite (Using INT_MAX 
# caused overflow problems)
NUM_MAX = 100000000.0
 
# # Given three collinear points p, q, r, 
# # the function checks if point q lies
# # on line segment 'pr'
# def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
#     if ((q[0] <= max(p[0], r[0])) &
#         (q[0] >= min(p[0], r[0])) &
#         (q[1] <= max(p[1], r[1])) &
#         (q[1] >= min(p[1], r[1]))):
#         return True
         
#     return False
 
# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
# def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
#     val = (((q[1] - p[1]) *
#             (r[0] - q[0])) -
#            ((q[0] - p[0]) *
#             (r[1] - q[1])))
            
#     if val == 0:
#         return 0
#     if val > 0:
#         return 1 # Collinear
#     else:
#         return 2 # Clock or counterclock

 
# def doIntersect(p1, q1, p2, q2):
     
#     # Find the four orientations needed for 
#     # general and special cases
#     o1 = orientation(p1, q1, p2)
#     o2 = orientation(p1, q1, q2)
#     o3 = orientation(p2, q2, p1)
#     o4 = orientation(p2, q2, q1)
 
#     # General case
#     if (o1 != o2) and (o3 != o4):
#         return True
     
#     # Special Cases
#     # p1, q1 and p2 are collinear and
#     # p2 lies on segment p1q1
#     if (o1 == 0) and (onSegment(p1, p2, q1)):
#         return True
 
#     # p1, q1 and p2 are collinear and
#     # q2 lies on segment p1q1
#     if (o2 == 0) and (onSegment(p1, q2, q1)):
#         return True
 
#     # p2, q2 and p1 are collinear and
#     # p1 lies on segment p2q2
#     if (o3 == 0) and (onSegment(p2, p1, q2)):
#         return True
 
#     # p2, q2 and q1 are collinear and
#     # q1 lies on segment p2q2
#     if (o4 == 0) and (onSegment(p2, q1, q2)):
#         return True
 
#     return False
 
# Returns true if the point p lies 
# inside the polygon[] with n vertices
#def is_inside_polygon(points:list, p:tuple) -> bool:
def is_inside_polygon(points, p) -> bool:   
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = Point(NUM_MAX, p.y)
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (line_segments_intersect_or_not.doIntersect(points[i],
                        points[next],
                        p, extreme)):
                             
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if line_segments_intersect_or_not.orientation(points[i], p,
                           points[next]) == 0:
                return line_segments_intersect_or_not.onSegment(points[i], p,
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
         
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)
 

def get_random_color_based_on_GP(name):
    if len(name)<3:
        return (0,0,0)
    else:
        return ((ord(name[0])*17)%125, (ord(name[1])*17)%125, (ord(name[2])*17)%125)
# Driver code
if __name__ == '__main__':
    my_str='jdyeb'
    #print(ord(my_str[0]))
    print(get_random_color_based_on_GP(my_str))
    #polygon1 = [ Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10) ]
    polygon1 = [ Point(0, 0), Point(10, 0), Point(10, 9), Point(5, 15), Point(0, 10) ]

    p = Point(-1, 15)#False
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')

    p = Point(-1, 0)#False
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')

    p = Point(-1, 10)#False, this place is wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')

    p = Point(-1, 10.5)#False, this place is wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')
     
    p = Point(20, 20)
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')
       
    p = Point(5, 5)
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')
 
    polygon2 = [ Point(0, 0), Point(5, 0), Point(5, 5), Point(3, 3) ]
     
    p = Point(3, 3)
    if (is_inside_polygon(points = polygon2, p = p)):
      print ('Yes')
    else:
      print ('No')
       
    p = Point(5, 1)
    if (is_inside_polygon(points = polygon2, p = p)):
      print ('Yes')
    else:
      print ('No')
       
    p = Point(8, 1)
    if (is_inside_polygon(points = polygon2, p = p)):
      print ('Yes')
    else:
      print ('No')
     
       

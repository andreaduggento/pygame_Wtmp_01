import pygame
import random
import string


lowercaseletters = string.ascii_lowercase

def randomname(lenght):
    string= ''.join(random.choice(lowercaseletters) for i in range(lenght))
    return string


def blitRotateCenteredRef(surf, image, pos, originPos, angle, display_width , display_height ):

    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
    
    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    """ include convertion to centered coordinates """
    rotated_image_center = (pos[0] - rotated_offset.x + display_width/2 , display_height/2 - pos[1] + rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)
  
    # draw rectangle around the image
    # pygame.draw.rect(surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()),2)


def blitRotateBottomLeftRef(surf, image, pos, originPos, angle, display_width , display_height ):

    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0]/2. , pos[1] - originPos[1]/2. ))
    offset_center_to_pivot = pygame.math.Vector2(pos.tolist()) - image_rect.center
    
    
    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    """ include convertion to centered coordinates """
    rotated_image_center = (pos[0] - rotated_offset.x , display_height - pos[1] + rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)
  
    # draw rectangle around the image
    #pygame.draw.rect(surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()),2)



"""Candidate presentation factors; annotation indices map explicitly to answer letters."""
from PIL import ImageDraw, ImageFont
from .edits import overlay_boxes


def numbered_boxes(image, boxes, symbols="1234"):
    image=image.copy(); draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=max(20,round(max(image.size)*.04)))
    for symbol,box in zip(symbols,boxes):
        draw.rectangle(box,outline="white",width=3)
        x,y=box[0],max(0,box[1]-font.size-6)
        bb=draw.textbbox((x+2,y),symbol,font=font)
        draw.rectangle((x,y,bb[2]+3,bb[3]+2),fill="black",outline="white",width=1)
        draw.text((x+2,y),symbol,font=font,fill="white")
    return image


def present(image, item, style="legacy"):
    if item['dataset']=='gqa':
        options=item['options'].copy();gold=item['gold']
        if style=='outer_rotation':
            options=options[-1:]+options[:-1];gold='ABCD'[('ABCD'.index(gold)+1)%4]
        elif style!='legacy':raise ValueError('Visual region relabeling is undefined for GQA color choices')
        return image,item['question'],options,gold
    boxes=item['boxes']; question=item['question']; options=item['options'].copy(); gold=item['gold']
    if style=='legacy': image=overlay_boxes(image,boxes)
    elif style=='numbered':
        image=numbered_boxes(image,boxes)
        question=question.replace('Select the box matching this description:','Which numbered region matches this description:')
        options=[f"region {i}" for i in range(1,5)]
    elif style=='outer_rotation':
        image=overlay_boxes(image,boxes); options=options[-1:]+options[:-1]; gold="ABCD"[("ABCD".index(gold)+1)%4]
    elif style=='visual_rotation':
        # A -> B, B -> C, C -> D, D -> A on the IMAGE; answer slots remain A-D.
        image=numbered_boxes(image,boxes,"BCDA"); gold="ABCD"[("ABCD".index(gold)+1)%4]
    else: raise ValueError(style)
    return image,question,options,gold

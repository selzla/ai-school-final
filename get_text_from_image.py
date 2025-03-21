from google.cloud import vision
import io

def get_orientation(vertices):
    start_x = vertices[0][0]
    start_y = vertices[0][1]
    end_x = vertices[2][0]
    end_y = vertices[2][1]
    if end_x >= start_x and end_y >= start_y:
        return 'up'
    elif end_x >= start_x and end_y <= start_y:
        return 'left'
    elif end_x <= start_x and end_y >= start_y:
        return 'right'
    elif end_x <= start_x and end_y <= start_y:
        return 'down'

def get_text_from_image(page):
    height = page.size[0]
    client = vision.ImageAnnotatorClient()

    buffer = io.BytesIO()
    page.save(buffer, 'JPEG')
    content = buffer.getvalue()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image,
                                             image_context={'language_hints':['en','ja']})
    document_text_data = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:

                for word in paragraph.words:
                    vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                    chars = []
                    confs = []
                    for symbol in word.symbols:
                        chars.append(symbol.text)
                        confs.append(symbol.confidence)
                    d = {'content': ''.join(chars),
                        'vertices': vertices,
                        'confs': confs,
                        'orientation': get_orientation(vertices)}

                    if len(d['content']) == len(confs) and abs(d['vertices'][2][1] - d['vertices'][0][1]) > height / 150:
                        document_text_data.append(d)

    if len(response.text_annotations) > 0:
        blob = response.text_annotations[0]
        blob_data = {}
        blob_vertices = [(vertex.x, vertex.y) for vertex in blob.bounding_poly.vertices]
        blob_data['content'] = blob.description
        blob_data['vertices'] = blob_vertices
        blob_data['confs'] = []
        blob_data['orientation'] = 'up'
        document_text_data = [blob_data] + document_text_data

    return document_text_data
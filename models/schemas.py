from marshmallow import Schema, fields, validate, ValidationError

class MoviesGenresPredictionSchema(Schema):
    plot = fields.Str(required=True)

class BatchSchema(Schema):
    batch = fields.List(fields.Nested(MoviesGenresPredictionSchema))


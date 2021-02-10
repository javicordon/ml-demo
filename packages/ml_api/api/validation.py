import typing as t

from marshmallow import Schema, fields
from marshmallow import ValidationError

from api import config


class InvalidInputError(Exception):
    """Invalid model input."""


SYNTAX_ERROR_FIELD_MAP = {
    #'3active_cch3_months': 'three_active_cch3_months',
}


class HouseDataRequestSchema(Schema):
    cl_unq_act_act_totalgastosfam = fields.Float(allow_none=True)
    cl_unq_act_act_totalgastosfam = fields.Float(allow_none=True)
    cl_unq_act_act_negociototalingresos = fields.Float(allow_none=True)
    cl_unq_act_act_totalbienes = fields.Float(allow_none=True)
    cl_unq_act_act_monto = fields.Float(allow_none=True)
    cl_unq_act_act_ptodestino  = fields.Str()
    cl_unq_act_act_flagpuedeescribir  = fields.Str()
    cl_unq_act_act_flagpuedeleer  = fields.Str()
    cl_unq_act_act_flagaccesovehicular  = fields.Str()
    cl_unq_act_act_flagaccesomensajeros  = fields.Str()
    cl_unq_act_act_flagtienegarage  = fields.Str()
    cl_unq_act_act_flagtienecomedor  = fields.Str()
    cl_unq_act_act_flagtieneagua  = fields.Str()
    cl_unq_act_act_flagtienerefrigerador  = fields.Str()
    cl_unq_act_act_flagtienelavadora  = fields.Str()
    qty_meses_desde_desembolso  = fields.Float()
    cl_unq_act_act_messolicitud  = fields.Float()
    cl_unq_act_act_fnacimiento_date  = fields.Str()


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""
    # set many=True to allow passing in a list
    schema = HouseDataRequestSchema(strict=True, many=True)

    # convert syntax error field names (beginning with numbers)
    for dict in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict[value] = dict[key]
            del dict[key]

    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    # convert syntax error field names back
    # this is a hack - never name your data
    # fields with numbers as the first letter.
    for dict in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict[key] = dict[value]
            del dict[value]

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

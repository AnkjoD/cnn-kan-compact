import pytest
import torch
from fair2026_lms_kan.cli_pipeline import jobs, _check_checkpoint


def test_priority_order_and_no_crop():
    assert jobs(['fastkan','linear','mlp'],[42,123]) == [
        ('fastkan',42,True),('linear',42,True),('mlp',42,True),('fastkan',42,False),
        ('fastkan',123,True),('linear',123,True),('mlp',123,True),('fastkan',123,False)]


def test_checkpoint_must_match_manifest_and_head(tmp_path):
    ckpt=tmp_path/'checkpoint.pt'
    torch.save({'metadata':{'head':'linear','seed':42,'use_lung_crop':True,
                            'manifest_sha256':'abc','model':{'backbone_size':'pico'}}},ckpt)
    _check_checkpoint(ckpt,'linear',42,True,'abc',{'backbone_size':'pico'})
    with pytest.raises(ValueError,match='incompatible'):
        _check_checkpoint(ckpt,'linear',42,True,'different',{'backbone_size':'pico'})

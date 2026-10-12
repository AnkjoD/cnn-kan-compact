import zipfile
import pytest
from fair2026_lms_kan.cli_prepare_data import _extract, download
from fair2026_lms_kan.cli_audit_manifest import audit


def test_download_validates_extracted_dataset(tmp_path, monkeypatch):
    from fair2026_lms_kan import cli_prepare_data as module
    monkeypatch.setattr(module.shutil, 'which', lambda _: '/usr/bin/kaggle')
    calls=[]
    def fake_run(command, check):
        calls.append(command)
        with zipfile.ZipFile(tmp_path/'kermany.zip','w') as archive:
            for split in ('train','val','test'):
                for label in ('NORMAL','PNEUMONIA'):
                    archive.writestr(f'chest_xray/{split}/{label}/x.jpeg',b'fake')
    monkeypatch.setattr(module.subprocess,'run',fake_run)
    assert download('kermany',tmp_path)
    assert not download('kermany',tmp_path)
    assert len(calls)==1
    assert calls[0][1:4] == ['datasets','download','paultimothymooney/chest-xray-pneumonia']


def test_download_rejects_incomplete_and_unsafe_archives(tmp_path,monkeypatch):
    from fair2026_lms_kan import cli_prepare_data as module
    monkeypatch.setattr(module.shutil,'which',lambda _: '/usr/bin/kaggle')
    def incomplete(command,check):
        with zipfile.ZipFile(tmp_path/'empty.zip','w') as archive:
            archive.writestr('README.txt','nothing')
    monkeypatch.setattr(module.subprocess,'run',incomplete)
    with pytest.raises(RuntimeError,match='expected folders'):
        download('kermany',tmp_path)
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as archive:
        archive.writestr('../escape.txt','bad')
    with pytest.raises(ValueError,match='Unsafe path'):
        _extract(tmp_path/'bad.zip',tmp_path)


def test_audit_rejects_patient_leakage(tmp_path):
    manifest=tmp_path/'m.csv'
    manifest.write_text('path,label,split,patient_id\na.png,0,train,p1\nb.png,1,val,p1\nc.png,0,test,p2\n')
    with pytest.raises(ValueError,match='patients appear'):
        audit(manifest)

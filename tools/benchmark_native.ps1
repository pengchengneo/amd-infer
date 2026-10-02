param(
 [Parameter(Mandatory=$true)][ValidateSet('fp32','default')][string]$Mode,
 [Parameter(Mandatory=$true)][string]$Runtime,
 [Parameter(Mandatory=$true)][string]$Model,
 [Parameter(Mandatory=$true)][string]$Cases,
 [Parameter(Mandatory=$true)][string]$Output,
 [string]$ExpectedManifest,
 [switch]$Worker
)
$ErrorActionPreference='Stop'
# Set MMVQ before creating the process which loads llama/ggml. Mutating env
# inside the DLL-loading process does not reliably change native CRT getenv.
if(!$Worker){
 if($Mode -eq 'fp32'){$env:GGML_VK_DISABLE_MMVQ='1'}else{Remove-Item Env:GGML_VK_DISABLE_MMVQ -ErrorAction SilentlyContinue}
 $child=@('-NoProfile','-File',$PSCommandPath,'-Worker','-Mode',$Mode,'-Runtime',$Runtime,'-Model',$Model,'-Cases',$Cases,'-Output',$Output)
 if($ExpectedManifest){$child+=@('-ExpectedManifest',$ExpectedManifest)}
 & (Get-Process -Id $PID).Path @child
 exit $LASTEXITCODE
}
if($Mode -eq 'fp32' -and $env:GGML_VK_DISABLE_MMVQ -ne '1'){throw 'FP32 flag was not inherited at startup'}
if($Mode -eq 'default' -and $env:GGML_VK_DISABLE_MMVQ){throw 'Default MMVQ flag must be unset at startup'}
$Runtime=(Resolve-Path -LiteralPath $Runtime).Path;$Model=(Resolve-Path -LiteralPath $Model).Path;$Cases=(Resolve-Path -LiteralPath $Cases).Path
if(Test-Path -LiteralPath $Output){throw 'Preserve existing benchmark output'}
$hashes=@{};foreach($file in Get-ChildItem -LiteralPath $Runtime -Filter '*.dll'){$hashes[$file.Name]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()}
if($ExpectedManifest){$expected=(Get-Content -Raw -LiteralPath $ExpectedManifest | ConvertFrom-Json).runtime_hashes;foreach($property in $expected.PSObject.Properties){if($hashes[$property.Name] -ne $property.Value){throw ('Pinned DLL changed: '+$property.Name)}}}
Add-Type -Path @((Join-Path $PSScriptRoot 'native-llama-quality.cs'),(Join-Path $PSScriptRoot 'phase13-native.cs'),(Join-Path $PSScriptRoot 'phase9-adl.cs'))
if(![NativeQuality]::SetDllDirectory($Runtime)){throw 'DLL directory failure'}
New-Item -ItemType Directory -Path $Output | Out-Null
@{utc=[DateTime]::UtcNow.ToString('o');mode=$Mode;runtime_hashes=$hashes;MMVQ_inherited=$env:GGML_VK_DISABLE_MMVQ;runtime=$Runtime;model=$Model;warm_tokens=256;outputs=64;repeats=3;context=512;batch=1;ubatch=1;kv='FP32';flash_attention=0;case_warmup=$true;observer_callback=$false;physical_reserve_bytes=2164260864;arithmetic=if($Mode -eq 'fp32'){'FP32 activation matvec'}else{'default MMVQ quantized activations'};pinned_manifest_verified=[bool]$ExpectedManifest} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Output 'launch.json') -Encoding utf8
$env:PHASE13_CAPTURE='0'
try{
 $paths=@(Get-ChildItem -LiteralPath $Cases -Filter '*.txt' | Sort-Object Name | ForEach-Object {$_.FullName})
 [Phase13Native]::Run($Runtime,$Model,$paths,$Output,3,256,64,$true)
 @{returncode=0;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'exit.json') -Encoding utf8
}catch{
 @{returncode=1;error=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'exit.json') -Encoding utf8
 throw
}finally{[NativeQuality]::SetDllDirectory($null) | Out-Null}

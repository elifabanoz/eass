rule EICAR_Test_File
{
    meta:
        confidence = "high"
        category = "test"
    strings:
        $prefix = { 58 35 4F 21 50 25 40 41 50 5B 34 5C 50 5A 58 35 34 28 50 5E 29 37 43 43 29 37 7D 24 }
        $marker = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*" ascii
    condition:
        filesize == 68 and $prefix at 0 and $marker at 28
}

rule Injection_API_Strings
{
    meta:
        confidence = "medium"
        category = "process-injection-indicators"
    strings:
        $alloc = "VirtualAllocEx" ascii wide
        $write = "WriteProcessMemory" ascii wide
        $thread = "CreateRemoteThread" ascii wide
    condition:
        uint16(0) == 0x5A4D and all of them
}


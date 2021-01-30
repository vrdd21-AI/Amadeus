package com.example.amadeus;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class AmadeusActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_amadeus);
    }

    public void ConnectClicked(View view){
        Log.d("test", "Connect");
        Intent intent = new Intent(this, AmadeusConnectActivity.class);
        startActivity(intent);
    }

    public void CancelClicked(View view){
        Log.d("test", "Cancel");
        finish();
    }
}
